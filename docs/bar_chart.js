
const barChartMain = () => {

    const data_location = './visualization_data/best_results.json';

    const svg = d3.select('#bar-chart-svg');
    const barChartGroup = svg.append('g');
    const barChartTitle = barChartGroup.append('text');
    const attentionModelBars = barChartGroup.append('g');
    const plainRNNModelBars = barChartGroup.append('g');
    const xAxisGroup = barChartGroup.append('g');
    const yAxisGroup = barChartGroup.append('g');
    const yAxisLabel = yAxisGroup.append('text');
    const toolTipGroup = svg.append('g');
    const toolTipBoundingBox = toolTipGroup
          .append('rect')
          .style('opacity', 0);
    
    const margin = {
        top: 80,
        bottom: 20,
        left: 120,
        right: 30,
    };

    const attentionFill = 'red';
    const attentionToolTipFill = '#ff99ad';
    const plainRNNFill = 'blue';
    const plainRNNToolTipFill = '#9cb3ff';
    const toolTipTransitionTime = 250;
    
    const render = (attention_data, plain_rnn_data) => {
        
        const plotContainer = document.getElementById('bar-chart');
        svg
            .attr('width', plotContainer.clientWidth)
            .attr('height', plotContainer.clientHeight);
        
        const svgWidth = parseFloat(svg.attr('width'));
        const svgHeight = parseFloat(svg.attr('height'));
        
        const innerWidth = svgWidth - margin.left - margin.right;
        const innerHeight = svgHeight - margin.top - margin.bottom;

        const labelSize = Math.min(20, innerWidth/40);
        barChartTitle
            .style('font-size', labelSize)
            .attr('x', innerWidth * 0.5)
            .attr('y', margin.top - labelSize * 2)
            .text('Best Performing Models');

        const nameForAttentionModel = datum => `Attention Model #${datum.model_index}`;
        const nameForPlainRNNModel = datum => `RNN Model #${datum.model_index}`;
        const attentionNames = attention_data.map(nameForAttentionModel);
        const plainRNNNames = plain_rnn_data.map(nameForPlainRNNModel);
        const modelNames = attentionNames.concat(plainRNNNames);
        
        const xScale = d3.scaleBand()
              .domain(modelNames)
              .range([0, innerWidth]);
        
        const yScale = d3.scaleLinear()
              .domain([0.455, 0.4])
              .range([0, innerHeight]);
        
        yAxisGroup.call(d3.axisLeft(yScale).tickSize(-innerWidth))
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
        yAxisGroup.selectAll('.tick line')
            .attr('x', margin.left - 10)
            .style('opacity', 0.1);
        yAxisGroup.selectAll('.tick text')
            .attr('transform', 'translate(-10.0, 0.0)');
        yAxisLabel
            .style('font-size', labelSize * 0.8)
            .attr('fill', 'black')
            .attr('transform', 'rotate(-90)')
            .attr('y', -60)
            .attr('x', -innerHeight/3)
            .text('Mean Cross Entropy Loss');
        
        const tickTextSize = Math.min(15, innerWidth/40);
        const mean = arr => arr.reduce((a,b) => a + b, 0) / arr.length;
        const tickTextAverageLength = mean(modelNames.map(name => name.length));
        xAxisGroup.call(d3.axisBottom(xScale).tickSize(-innerHeight))
            .attr('transform', `translate(${margin.left}, ${margin.top+innerHeight})`);
        xAxisGroup.selectAll('.tick line')
            .remove();
        xAxisGroup.selectAll('.tick text')
            .attr('transform', `translate(${-tickTextSize/2}, ${- tickTextAverageLength * tickTextSize * 0.4}) rotate(-90)`)
            .style('color', '#fff')
            .style('font-size', tickTextSize);
        
        const updateToolTip = (mouseX, mouseY, desiredOpacity, datum, backgroundColor) => {
            toolTipGroup.selectAll('g').remove();
            const toolTipTextLines = [
                `Number of Epochs: ${datum.number_of_epochs}`,
                `Batch Size: ${datum.batch_size}`,
                `Vocab Size: ${datum.vocab_size}`,
                `Pretrained Embedding: ${datum.pre_trained_embedding_specification}`,
                `LSTM Hidden Size: ${datum.encoding_hidden_size}`,
                `Number of LSTM Layers: ${datum.number_of_encoding_layers}`
            ];
            if (datum.attention_intermediate_size) {
                toolTipTextLines.push(
                    `Attention Intermediate Size: ${datum.attention_intermediate_size}`,
                    `Number of Attention Heads: ${datum.number_of_attention_heads}`
                );
            }
            toolTipTextLines.push(
                `Dropout Probability: ${datum.dropout_probability}`,
                `Test Loss: ${datum.test_loss}`,
                `Test Accuracy: ${datum.test_accuracy}`,
                `Number of Parameters: ${datum.number_of_parameters}`
            );
            const ephemeralTextLinesGroup = toolTipGroup.append('g');
            toolTipTextLines.forEach((textLine, textLineIndex) => {
                ephemeralTextLinesGroup
                    .append('text')
                    .style('font-size', labelSize)
                    .attr('class', 'displayed-text')
                    .attr('dx', labelSize)
                    .attr('dy', `${(1+textLineIndex) * 1.2}em`)
                    .html(textLine);
            });
            const ephemeralTextLinesGroupBBox = ephemeralTextLinesGroup.node().getBBox();
            const toolTipBoundingBoxWidth = ephemeralTextLinesGroupBBox.width + 2 * labelSize;
            const toolTipBoundingBoxHeight = ephemeralTextLinesGroupBBox.height + labelSize;

            const rightLimit = margin.left + innerWidth;
            const mouseCloserToRight = mouseX - margin.left > rightLimit - mouseX;
            const toolTipX = desiredOpacity === 0 ? -svgWidth : (mouseCloserToRight ? margin.left + labelSize : rightLimit - labelSize - toolTipBoundingBoxWidth);
            
            const bottomLimit = margin.top + innerHeight;
            const mouseCloserToBottom = mouseY - margin.top > bottomLimit - mouseY;
            const toolTipY = desiredOpacity === 0 ? -svgHeight : (mouseCloserToBottom ? margin.top + labelSize : bottomLimit - labelSize - toolTipBoundingBoxHeight);
            
            toolTipBoundingBox
                .attr('x', toolTipX)
                .attr('y', toolTipY)
                .style('stroke-width', 1)
                .style('stroke', 'black')
                .style('fill', backgroundColor)
                .attr('width', toolTipBoundingBoxWidth)
                .attr('height', toolTipBoundingBoxHeight);
            ephemeralTextLinesGroup.selectAll('*')
                .attr('x', toolTipX)
                .attr('y', toolTipY);
            const elementsSelection = toolTipGroup.selectAll('*');
            elementsSelection
                .transition()
                .duration(toolTipTransitionTime)
                .style('opacity', desiredOpacity);
        };
        
        attentionModelBars.selectAll('rect').data(attention_data)
            .remove();
        attentionModelBars.selectAll('rect').data(attention_data)
            .enter()
            .append('rect')
            .on('mouseenter', function(datum) {
                const [mouseX, mouseY] = d3.mouse(this);
                updateToolTip(mouseX, mouseY, 1, datum, attentionToolTipFill);
            })
            .on('mouseleave', function(datum) {
                const [mouseX, mouseY] = d3.mouse(this);
                updateToolTip(mouseX, mouseY, 0, datum, attentionToolTipFill);
            })
            .attr('y', datum => margin.top+yScale(datum['test_loss']))
            .attr('x', datum => xScale(nameForAttentionModel(datum))+margin.left)
            .attr('width', xScale.bandwidth())
            .attr('height', datum => innerHeight-yScale(datum['test_loss']))
            .attr('fill', attentionFill);
        
        plainRNNModelBars.selectAll('rect').data(plain_rnn_data)
            .remove();
        plainRNNModelBars.selectAll('rect').data(plain_rnn_data)
            .enter()
            .append('rect')
            .on('mouseenter', function(datum) {
                const [mouseX, mouseY] = d3.mouse(this);
                updateToolTip(mouseX, mouseY, 1, datum, plainRNNToolTipFill);
            })
            .on('mouseleave', function(datum) {
                const [mouseX, mouseY] = d3.mouse(this);
                updateToolTip(mouseX, mouseY, 0, datum, plainRNNToolTipFill);
            })
            .attr('y', datum => margin.top+yScale(datum['test_loss']))
            .attr('x', datum => xScale(nameForPlainRNNModel(datum))+margin.left)
            .attr('width', xScale.bandwidth())
            .attr('height', datum => innerHeight-yScale(datum['test_loss']))
            .attr('fill', plainRNNFill);
        
    };
    
    d3.json(data_location)
        .then(data => {
            let extract_data = (datum, datumIndex) => {
                return {
                    model_index: datumIndex, 
                    number_of_epochs: parseInt(datum.number_of_epochs),
                    batch_size: parseInt(datum.batch_size),
                    vocab_size: parseInt(datum.vocab_size),
                    pre_trained_embedding_specification: datum.pre_trained_embedding_specification,
                    encoding_hidden_size: parseInt(datum.encoding_hidden_size),
                    number_of_encoding_layers: parseInt(datum.number_of_encoding_layers),
                    attention_intermediate_size: parseInt(datum.attention_intermediate_size),
                    number_of_attention_heads: parseInt(datum.number_of_attention_heads),
                    dropout_probability: parseFloat(datum.dropout_probability),
                    final_representation: datum.final_representation,
                    test_loss: parseFloat(datum.test_loss),
                    test_accuracy: parseFloat(datum.test_accuracy),
                    number_of_parameters: parseInt(datum.number_of_parameters),
                };
            };
            let attention_data = data['attention'].map(extract_data);
            let plain_rnn_data = data['plain_rnn'].map(extract_data);
            const redraw = () => { 
                render(attention_data, plain_rnn_data);
            };
            redraw();
            window.addEventListener('resize', redraw);
        }).catch(err => {
            console.error(err.message);
            return;
        });


};

barChartMain();
